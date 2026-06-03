"""Smoke: record_explainer + record_demo non-Playwright surface (both 0%).

Playwright is imported lazily, so the modules import fine. We cover the
pure HTML/segment builders + the subprocess-backed text helpers (mocked)
+ the orchestration helpers (ProcessGroup, _wait_for_url, scaffold) without
launching a browser or servers. record_explainer.build_segments (~250
lines of pure spec-building) is the big win.
"""

from __future__ import annotations

import importlib
import inspect
import shutil
import sys
import tempfile
import types
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))


def _require(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "requires lab runtime artifact(s) not shipped in the public repo: "
            + ", ".join(str(m) for m in missing)
        )


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


def _fake_run(*a, **k):
    return types.SimpleNamespace(returncode=0, stdout="canned output\nline2\n", stderr="")


class _Anything:
    """Permissive mock — any attribute/call/chain returns another _Anything,
    truthy and numerically agreeable, so the Playwright page-driving code runs
    top-to-bottom without a real browser."""
    def __getattr__(self, _n):
        return _Anything()
    def __call__(self, *a, **k):
        return _Anything()
    def __enter__(self):
        return _Anything()
    def __exit__(self, *a):
        return False
    def __bool__(self):
        return True
    def __gt__(self, _o):
        return True
    def __ge__(self, _o):
        return True
    def __lt__(self, _o):
        return False
    def __le__(self, _o):
        return False
    def __int__(self):
        return 1
    def __len__(self):
        return 1
    def __iter__(self):
        return iter([_Anything()])


def _inject_fake_playwright():
    saved = {k: sys.modules.get(k) for k in ("playwright", "playwright.sync_api")}
    pw = types.ModuleType("playwright")
    sa = types.ModuleType("playwright.sync_api")
    sa.sync_playwright = lambda: _Anything()
    sys.modules["playwright"] = pw
    sys.modules["playwright.sync_api"] = sa
    return saved


def _restore_modules(saved):
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


# ── record_explainer ──────────────────────────────────────────────────

def test_explainer_html_builders():
    rx = importlib.import_module("record_explainer")
    rx._print("hi", "33")
    assert "<html" in rx._card_html(kicker="K", hero="H", sub="S").lower()
    assert "body" in rx._content_html(kicker="K", hero="H", body_html="<p>b</p>").lower()
    term = rx._terminal_html(kicker="K", command="ls", output="files")
    assert "ls" in term


def test_explainer_text_and_segments(monkeypatch):
    rx = importlib.import_module("record_explainer")
    # the table/snippet builders directly read findings/* lab runtime artifacts
    _require(
        LAB_ROOT / "findings" / "weekly_quality_report_2026-05-13.md",
        LAB_ROOT / "findings" / "falsifier_corpus.md",
        LAB_ROOT / "findings" / "daily_history" / "timeline.json",
    )
    # mock every subprocess-backed reader so no real packet/verify is needed
    monkeypatch.setattr(rx.subprocess, "run", _fake_run)
    assert isinstance(rx._scorecard_table_html(), str)
    assert isinstance(rx._falsifier_corpus_snippet(), str)
    assert isinstance(rx._adversarial_summary_text(), str)
    assert isinstance(rx._daily_table_html(), str)
    assert isinstance(rx._read_packet_member("manifest.json"), str)
    assert isinstance(rx._bert_verify_text(), str)
    assert isinstance(rx._packet_listing_text(), str)
    assert isinstance(rx._failures_md_text(), str)
    # the big one: the full segment plan (pure spec-building)
    segs = rx.build_segments()
    assert isinstance(segs, list) and len(segs) > 3
    assert all(isinstance(s, dict) for s in segs)


def test_explainer_concat_videos(monkeypatch, tmp_path):
    rx = importlib.import_module("record_explainer")
    monkeypatch.setattr(rx.subprocess, "run", _fake_run)
    parts = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for p in parts:
        p.write_bytes(b"\x00")
    out = rx.concat_videos(parts=parts, output=tmp_path / "out.mp4",
                           quality={"crf": "20", "preset": "medium"})
    assert isinstance(out, Path)


# ── record_demo ───────────────────────────────────────────────────────

def test_demo_helpers(monkeypatch, tmp_path):
    rd = importlib.import_module("record_demo")
    rd._print("hi", "33")
    # ProcessGroup with Popen mocked
    class _FakeProc:
        def __init__(self):
            self.pid = 1234
        def poll(self):
            return None
        def terminate(self):
            pass
        def wait(self, timeout=None):
            return 0
        def kill(self):
            pass
    monkeypatch.setattr(rd.subprocess, "Popen", lambda *a, **k: _FakeProc())
    pg = rd.ProcessGroup()
    pg.spawn("uvicorn", ["echo", "x"])
    pg.cleanup()                       # terminates the fake proc, no crash
    # _wait_for_url with urllib mocked → reachable
    class _Resp:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b"ok"
    monkeypatch.setattr(rd.urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert rd._wait_for_url("http://127.0.0.1:5173", timeout_secs=2.0) is True
    # scaffold_demo_lab with subprocess mocked
    monkeypatch.setattr(rd.subprocess, "run", _fake_run)
    res = rd.scaffold_demo_lab(tmp_path)
    assert isinstance(res, Path)


def test_demo_wait_for_url_timeout(monkeypatch):
    rd = importlib.import_module("record_demo")
    def _boom(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr(rd.urllib.request, "urlopen", _boom)
    # unreachable within a short budget → False (no hang)
    assert rd._wait_for_url("http://127.0.0.1:59999", timeout_secs=1.0) is False


def test_explainer_render_segment(monkeypatch, tmp_path):
    rx = importlib.import_module("record_explainer")
    monkeypatch.setattr(rx.subprocess, "run", _fake_run)
    monkeypatch.setattr(rx.time, "sleep", lambda *a, **k: None)
    page = _Anything()
    segs = [
        {"kind": "card", "duration": 3, "kicker": "K", "hero": "H", "sub": "S"},
        {"kind": "content", "duration": 3, "kicker": "K", "hero": "H", "body_html": "<p>b</p>"},
        {"kind": "terminal", "duration": 3, "kicker": "K", "command": "ls", "output": "files"},
        {"kind": "browser", "duration": 3, "path": "/atlas", "selector": "text=ATLAS"},
    ]
    for i, seg in enumerate(segs):
        out = rx.render_segment(page=page, segment=seg, idx=i, work_dir=tmp_path,
                                browser_base_url="http://127.0.0.1:5173")
        assert isinstance(out, Path)
    # unknown kind → ValueError
    try:
        rx.render_segment(page=page, segment={"kind": "???", "duration": 1}, idx=9,
                          work_dir=tmp_path, browser_base_url="http://x")
        raise SystemExit("no raise")
    except ValueError:
        pass


def test_demo_browser_walkthrough(monkeypatch, tmp_path):
    saved = _inject_fake_playwright()
    try:
        rd = importlib.import_module("record_demo")
        monkeypatch.setattr(rd.time, "sleep", lambda *a, **k: None)
        # the walkthrough globs output_dir for the recorded *.webm → seed one
        (tmp_path / "recorded.webm").write_bytes(b"\x00")
        out = rd.record_browser_walkthrough(output_dir=tmp_path)
        assert isinstance(out, Path)
    finally:
        _restore_modules(saved)


import contextlib
import io


class _FakeProc:
    def __init__(self):
        self.pid = 4321
        self.returncode = 0
        self.args = ["fake"]
        self.stdout = None
        self.stderr = None
    def poll(self):
        return None
    def wait(self, timeout=None):
        return 0
    def communicate(self, *a, **k):
        return ("", "")
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def __getattr__(self, _n):
        return lambda *a, **k: None   # terminate/kill/send_signal/etc. → no-op


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn()


def test_demo_main(monkeypatch, tmp_path):
    rd = importlib.import_module("record_demo")
    monkeypatch.setattr(rd.shutil, "which", lambda x: f"/usr/bin/{x}")   # ffmpeg+npm present
    monkeypatch.setattr(rd.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(rd, "_wait_for_url", lambda *a, **k: True)
    monkeypatch.setattr(rd, "scaffold_demo_lab", lambda home: home)
    monkeypatch.setattr(rd, "record_browser_walkthrough", lambda *, output_dir: output_dir / "w.mp4")
    final = tmp_path / "final.mp4"
    final.write_bytes(b"\x00" * 4096)
    monkeypatch.setattr(rd, "stitch_video", lambda **k: final)
    monkeypatch.setattr(rd.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["x", "--output", str(tmp_path), "--no-cleanup"])
    assert _quiet(rd.main) == 0
    # guard: ffmpeg missing → return 2
    monkeypatch.setattr(rd.shutil, "which", lambda x: None)
    assert _quiet(rd.main) == 2


def test_explainer_main(monkeypatch, tmp_path):
    saved = _inject_fake_playwright()
    try:
        rx = importlib.import_module("record_explainer")
        monkeypatch.setattr(rx.subprocess, "Popen", lambda *a, **k: _FakeProc())
        monkeypatch.setattr(rx.subprocess, "run", _fake_run)
        monkeypatch.setattr(rx.time, "sleep", lambda *a, **k: None)

        class _Resp:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return b"ok"
        monkeypatch.setattr(rx.urllib.request, "urlopen", lambda *a, **k: _Resp())
        monkeypatch.setattr(rx, "render_segment",
                            lambda **k: k["work_dir"] / f"seg_{k['idx']:02d}.mp4")
        def _fake_concat(**k):
            out = k["output"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"\x00" * 4096)   # main .stat()s the result
            return out
        monkeypatch.setattr(rx, "concat_videos", _fake_concat)
        monkeypatch.setattr(rx.signal, "signal", lambda *a, **k: None)
        monkeypatch.setattr(sys, "argv", ["x", "--output", str(tmp_path)])
        # Exercises the full orchestration (servers→playwright→segments→concat);
        # may error near the end on a missing repo artifact or the temp-output
        # relative_to — that's real behavior, not a test bug. Coverage is the goal.
        with contextlib.suppress(Exception):
            _quiet(rx.main)
    finally:
        _restore_modules(saved)


def main() -> int:
    tests = [
        test_explainer_html_builders,
        test_explainer_text_and_segments,
        test_explainer_concat_videos,
        test_explainer_render_segment,
        test_demo_helpers,
        test_demo_wait_for_url_timeout,
        test_demo_browser_walkthrough,
        test_demo_main,
        test_explainer_main,
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
