"""Smoke + TDD: `bert project` CLI (Sprint 7 #31).

`bert project list / status / finalize / verify / approve` — a thin CLI family
over production backends (bert_lab MCP handlers, skill_runner, bert_verify,
proposal_activate). project == lab (same ~/.bert/labs scope). The CLI is wired
into lab.py main() via project_cli.build_subparser + dispatch.

Backends are stubbed so the CLI logic (parse -> dispatch -> rc) is proven
network-free; the backends have their own suites.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

from tools import project_cli  # noqa: E402


def _parse(argv):
    parser = argparse.ArgumentParser(prog="bert")
    sub = parser.add_subparsers(dest="cmd")
    project_cli.build_subparser(sub)
    return parser.parse_args(argv)


# ── parser ───────────────────────────────────────────────────────────


def test_parser_list_and_status():
    a = _parse(["project", "list"])
    assert a.cmd == "project" and a.project_cmd == "list"
    b = _parse(["project", "status", "mylab"])
    assert b.project_cmd == "status" and b.lab == "mylab"


def test_parser_finalize_requires_objective_output():
    a = _parse(["project", "finalize", "mylab", "--objective", "Q", "--output", "out.md"])
    assert a.project_cmd == "finalize" and a.objective == "Q" and a.output == "out.md"


def test_parser_verify_and_approve():
    a = _parse(["project", "verify", "p1.tar.gz", "p2.tar.gz"])
    assert a.project_cmd == "verify" and a.packets == ["p1.tar.gz", "p2.tar.gz"]
    b = _parse(["project", "approve", "tool-x-abc"])
    assert b.project_cmd == "approve" and b.proposal_id == "tool-x-abc"


# ── command handlers (backends stubbed) ──────────────────────────────


def test_cmd_list_invokes_backend(monkeypatch):
    calls = {}
    from tools.mcp import bert_lab
    monkeypatch.setattr(bert_lab, "_t_lab_list", lambda a: calls.setdefault("c", True) or {"labs": []})
    rc = project_cli.cmd_list(SimpleNamespace(json=True))
    assert rc == 0 and calls["c"]


def test_cmd_status_invokes_backend(monkeypatch):
    from tools.mcp import bert_lab
    monkeypatch.setattr(bert_lab, "_t_lab_status", lambda a: {"lab": a["lab"], "cycle": 3})
    rc = project_cli.cmd_status(SimpleNamespace(lab="mylab", json=True))
    assert rc == 0


def test_cmd_finalize_runs_skill(monkeypatch):
    from core import skill_runner
    from tools.mcp import bert_lab
    monkeypatch.setattr(bert_lab, "_resolve_lab", lambda n: Path("/tmp/lab"))
    seen = {}

    def fake_run(name, args, *, lab_path=None):
        seen["name"] = name
        seen["args"] = args
        return {"ok": True, "outputs": {"grade": "B", "ready": True}}

    monkeypatch.setattr(skill_runner, "run_skill", fake_run)
    rc = project_cli.cmd_finalize(SimpleNamespace(
        lab="mylab", objective="Q", output="out.md", json=True))
    assert rc == 0
    assert seen["name"] == "finalize_project"
    assert seen["args"]["objective"] == "Q" and seen["args"]["output_path"] == "out.md"


def test_cmd_finalize_unknown_lab_rc1(monkeypatch):
    from tools.mcp import bert_lab
    monkeypatch.setattr(bert_lab, "_resolve_lab", lambda n: None)
    rc = project_cli.cmd_finalize(SimpleNamespace(
        lab="nope", objective="Q", output="out.md", json=False))
    assert rc == 1


def test_cmd_finalize_failed_skill_rc1(monkeypatch):
    from core import skill_runner
    from tools.mcp import bert_lab
    monkeypatch.setattr(bert_lab, "_resolve_lab", lambda n: Path("/tmp/lab"))
    monkeypatch.setattr(skill_runner, "run_skill",
                        lambda *a, **k: {"ok": False, "errors": ["boom"]})
    rc = project_cli.cmd_finalize(SimpleNamespace(
        lab="x", objective="Q", output="o.md", json=False))
    assert rc == 1


def test_cmd_verify_delegates(monkeypatch):
    from tools import bert_verify
    seen = {}
    monkeypatch.setattr(bert_verify, "run",
                        lambda packets, **kw: seen.update(p=packets) or 0)
    rc = project_cli.cmd_verify(SimpleNamespace(
        packets=["a.tar.gz"], chain=False, no_color=True))
    assert rc == 0 and seen["p"] == ["a.tar.gz"]


def test_cmd_approve_routes(monkeypatch):
    from core import proposal_activate
    monkeypatch.setattr(proposal_activate, "activate",
                        lambda pid, **kw: {"ok": True, "kind": "tool", "name": "x"})
    rc = project_cli.cmd_approve(SimpleNamespace(proposal_id="tool-x-abc", json=True))
    assert rc == 0


def test_cmd_approve_failure_rc1(monkeypatch):
    from core import proposal_activate
    monkeypatch.setattr(proposal_activate, "activate",
                        lambda pid, **kw: {"ok": False, "error": "nope"})
    rc = project_cli.cmd_approve(SimpleNamespace(proposal_id="weird", json=False))
    assert rc == 1


def test_dispatch_routes_to_handlers(monkeypatch):
    hit = {}
    for name in ("cmd_list", "cmd_status", "cmd_finalize", "cmd_verify", "cmd_approve"):
        monkeypatch.setattr(project_cli, name,
                            (lambda n: (lambda args: hit.update({n: True}) or 0))(name))
    for sub, extra in [("list", {}), ("status", {"lab": "l"}),
                       ("finalize", {"lab": "l", "objective": "o", "output": "f"}),
                       ("verify", {"packets": ["p"]}), ("approve", {"proposal_id": "i"})]:
        args = SimpleNamespace(project_cmd=sub, json=False, **extra)
        assert project_cli.dispatch(args) == 0
    assert set(hit) == {"cmd_list", "cmd_status", "cmd_finalize", "cmd_verify", "cmd_approve"}


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
    import inspect
    tests = [
        test_parser_list_and_status,
        test_parser_finalize_requires_objective_output,
        test_parser_verify_and_approve,
        test_cmd_list_invokes_backend,
        test_cmd_status_invokes_backend,
        test_cmd_finalize_runs_skill,
        test_cmd_finalize_unknown_lab_rc1,
        test_cmd_finalize_failed_skill_rc1,
        test_cmd_verify_delegates,
        test_cmd_approve_routes,
        test_cmd_approve_failure_rc1,
        test_dispatch_routes_to_handlers,
    ]
    mp = _MP()
    for t in tests:
        try:
            if "monkeypatch" in inspect.signature(t).parameters:
                t(mp)
            else:
                t()
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
