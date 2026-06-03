"""Smoke: tools/bert_cli.py — the bert CLI (was 16%).

build_parser() is parsed for every subcommand (covers all argparse wiring);
_resolve_lab runs against a temp LABS_DIR; each cmd_* runs with the MCP
_t_* delegates + subprocess.run monkeypatched (json + human + error
branches). cmd_memory_ingest runs for real against a temp code_repo lab.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import shutil
import sys
import tempfile
import types
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

import tools.bert_cli as cli  # noqa: E402
import tools.mcp.bert_lab as mcp  # noqa: E402


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


def _ns(**kw):
    base = {"json": False, "prefix": None}
    base.update(kw)
    return argparse.Namespace(**base)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def test_build_parser_all_subcommands():
    ap = cli.build_parser()
    cases = [
        ["lab", "list"], ["lab", "list", "--prefix", "test"],
        ["lab", "start", "mylab", "a mission", "--archetype", "research", "--no-llm"],
        ["lab", "status", "mylab"],
        ["lab", "cycle", "mylab", "--no-claude"],
        ["lab", "reshape", "mylab", "primary_work=audit"],
        ["memory", "search", "mylab", "query terms", "-k", "3", "--method", "vector"],
        ["memory", "ingest", "mylab", "/some/path"],
        ["packet", "export", "mylab", "--cycle", "5"],
        ["packet", "verify", "/p.tar.gz", "--fetch-rekor"],
        ["doctor"], ["dashboard"],
    ]
    for argv in cases:
        ns = ap.parse_args(argv)
        assert hasattr(ns, "func")


def test_resolve_lab(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "LABS_DIR", tmp_path)
    (tmp_path / "mylab").mkdir()
    assert cli._resolve_lab("mylab") == tmp_path / "mylab"
    assert cli._resolve_lab("nonexistent") is None
    assert cli._resolve_lab("") is None
    assert cli._resolve_lab(str(tmp_path / "mylab")) == tmp_path / "mylab"  # absolute


def test_cmd_lab_list(monkeypatch):
    monkeypatch.setattr(mcp, "_t_lab_list", lambda a: {
        "labs": [{"lab": "x", "last_cycle": 1, "findings_count": 2, "events_total": 3}],
        "labs_dir": "/tmp/labs"})
    assert _quiet(cli.cmd_lab_list, _ns(prefix=None)) == 0
    assert _quiet(cli.cmd_lab_list, _ns(prefix="x", json=True)) == 0


def test_cmd_lab_status(monkeypatch):
    monkeypatch.setattr(mcp, "_t_lab_status", lambda a: {
        "ok": True, "lab": "x", "path": "/p", "mission": "m" * 200,
        "last_cycle": 1, "findings_count": 2, "events_total": 3})
    assert _quiet(cli.cmd_lab_status, _ns(name="x")) == 0
    assert _quiet(cli.cmd_lab_status, _ns(name="x", json=True)) == 0
    monkeypatch.setattr(mcp, "_t_lab_status", lambda a: {"ok": False, "error": "no lab"})
    assert _quiet(cli.cmd_lab_status, _ns(name="x")) == 3


def test_cmd_lab_start(monkeypatch):
    monkeypatch.setattr(mcp, "_t_lab_start", lambda a: {
        "ok": True, "lab": "x", "path": "/p",
        "profile": {"domain": "d", "primary_work": "w", "data_shape": "s"},
        "schema": {"rule_id": "r1"}, "scaffolded_knowledge_files": ["k.md"], "next": "go"})
    assert _quiet(cli.cmd_lab_start, _ns(name="x", mission="m", archetype=None, no_llm=True)) == 0
    monkeypatch.setattr(mcp, "_t_lab_start", lambda a: {"ok": False, "error": "exists"})
    assert _quiet(cli.cmd_lab_start, _ns(name="x", mission="m", archetype=None, no_llm=True)) == 3


def test_cmd_lab_cycle(monkeypatch):
    monkeypatch.setattr(mcp, "_t_lab_cycle", lambda a: {
        "ok": True, "cycles_before": 1, "cycles_after": 3, "findings_delta": 2,
        "events_delta": 9, "elapsed_secs": 12, "saturation": {"saturated": True}})
    assert _quiet(cli.cmd_lab_cycle, _ns(name="x", budget=None, no_claude=True)) == 0
    monkeypatch.setattr(mcp, "_t_lab_cycle", lambda a: {"ok": False, "error": "boom"})
    assert _quiet(cli.cmd_lab_cycle, _ns(name="x", budget="auto", no_claude=True)) == 1


def test_cmd_lab_reshape(monkeypatch):
    assert _quiet(cli.cmd_lab_reshape, _ns(name="x", updates=["bad_no_equals"])) == 2
    monkeypatch.setattr(mcp, "_t_lab_reshape", lambda a: {"ok": True})
    assert _quiet(cli.cmd_lab_reshape, _ns(name="x", updates=["primary_work=audit"])) == 0


def test_cmd_memory_search(monkeypatch):
    monkeypatch.setattr(mcp, "_t_memory_search", lambda a: {
        "ok": True, "method": "vector",
        "results": [{"path": "f.md", "snippet": "a snippet"}]})
    assert _quiet(cli.cmd_memory_search, _ns(name="x", query="q", k=3, method="vector")) == 0
    assert _quiet(cli.cmd_memory_search, _ns(name="x", query="q", k=None, method="hybrid", json=True)) == 0
    monkeypatch.setattr(mcp, "_t_memory_search", lambda a: {"ok": False, "error": "no idx"})
    assert _quiet(cli.cmd_memory_search, _ns(name="x", query="q", k=5, method="hybrid")) == 1


def test_cmd_memory_ingest_real(monkeypatch, tmp_path):
    import yaml
    monkeypatch.setattr(cli, "LABS_DIR", tmp_path)
    lab = tmp_path / "codelab"
    lab.mkdir()
    (lab / "lab.yaml").write_text(yaml.safe_dump({"mission_profile": {"data_shape": "code_repo"}}))
    src = lab / "m.py"
    src.write_text("def f():\n    return 1\n")
    assert _quiet(cli.cmd_memory_ingest, _ns(name="codelab", source=str(src))) == 0
    # lab not found
    assert _quiet(cli.cmd_memory_ingest, _ns(name="ghost", source="x")) == 3


def test_cmd_packet_export(monkeypatch):
    monkeypatch.setattr(mcp, "_t_packet_export", lambda a: {"ok": True, "packet": "/p.tar.gz"})
    assert _quiet(cli.cmd_packet_export, _ns(name="x", cycle=None)) == 0
    monkeypatch.setattr(mcp, "_t_packet_export", lambda a: {"ok": False})
    assert _quiet(cli.cmd_packet_export, _ns(name="x", cycle=5)) == 1


def test_cmd_subprocess_wrappers(monkeypatch):
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0))
    assert _quiet(cli.cmd_packet_verify, _ns(path="/p.tar.gz", fetch_rekor=False)) == 0
    assert _quiet(cli.cmd_packet_verify, _ns(path="/p.tar.gz", fetch_rekor=True)) == 0
    assert _quiet(cli.cmd_doctor, _ns()) == 0
    assert _quiet(cli.cmd_dashboard, _ns()) == 0


def test_main_dispatch(monkeypatch):
    monkeypatch.setattr(mcp, "_t_lab_list", lambda a: {"labs": [], "labs_dir": "/t"})
    assert _quiet(cli.main, ["bert", "lab", "list"]) == 0


def main() -> int:
    tests = [
        test_build_parser_all_subcommands,
        test_resolve_lab,
        test_cmd_lab_list,
        test_cmd_lab_status,
        test_cmd_lab_start,
        test_cmd_lab_cycle,
        test_cmd_lab_reshape,
        test_cmd_memory_search,
        test_cmd_memory_ingest_real,
        test_cmd_packet_export,
        test_cmd_subprocess_wrappers,
        test_main_dispatch,
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
