"""Smoke + TDD: core/skill_runner.py + the lab_finalize MCP trigger (task #69).

skill_executor is a pure engine; nothing in production built the
ExecutionContext it needs. run_skill() is that production driver — it registers
the tool suite, loads the skill registry, and runs a named skill against the
REAL tool_registry via make_invoker(). The MCP-first trigger is a `lab_finalize`
tool on the bert_lab server that calls run_skill("finalize_project", ...).

run_skill is tested end-to-end (stub provider, temp lab). The MCP tool is tested
for registration + that it routes to run_skill and shapes the result (run_skill
spied), decoupled from provider + lab resolution.
"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import quality, skill_runner  # noqa: E402
from core.types import ProviderResponse  # noqa: E402


@pytest.fixture
def tmp(tmp_path):
    """Alias for pytest's tmp_path so the e2e test (which the file's own
    main() runner drives with a self-managed temp dir) is also collectable
    and runnable under bare pytest."""
    return tmp_path


def _stub_provider(mp, *, judge=4):
    from core import provider as prov

    def fake(provider_name, messages, **kw):
        sp = messages[0]["content"].lower()
        if "artifact synthesizer" in sp:
            body = {"body": "# Final\n\nClaim[^0].\n\n[^0]: r1.md",
                    "citations_used": 1, "uncited_evidence": []}
        elif "gap auditor" in sp:
            body = {"gaps_md": "# Gaps\n- one benchmark", "gap_count": 1,
                    "unanswered_questions": ["edge cases?"], "honest_score": 0.8}
        else:
            body = dict.fromkeys(quality.DIMENSIONS, judge)
            body["rationale"] = "ok"
        return ProviderResponse(text=json.dumps(body), tool_calls=[],
                                finish_reason="stop", usage_prompt_tokens=10,
                                usage_completion_tokens=10, usage_thinking_tokens=0,
                                usage_cached_tokens=0, model="stub",
                                provider=provider_name, elapsed_ms=1)

    mp.setattr(prov, "call", fake)


def test_run_skill_executes_finalize_e2e(monkeypatch, tmp):
    # The production driver runs finalize_project against the REAL registry.
    _stub_provider(monkeypatch, judge=4)
    (tmp / "findings").mkdir()
    (tmp / "findings" / "r1.md").write_text("# R1\nbert beats BM25")
    res = skill_runner.run_skill(
        "finalize_project",
        {"objective": "Audit retrieval", "output_path": "final.md"},
        lab_path=tmp)
    assert res["ok"], res.get("errors")
    assert res["outputs"]["grade"] == "B"
    assert res["outputs"]["ready"] is True
    assert (tmp / "final.md").exists()


def test_run_skill_unknown_skill():
    res = skill_runner.run_skill("does_not_exist", {})
    assert res["ok"] is False and "not found" in res["error"]


def test_lab_finalize_mcp_tool_registered():
    from tools.mcp import bert_lab
    srv = bert_lab.make_server()
    assert "lab_finalize" in srv.tools, f"lab_finalize missing: {sorted(srv.tools)}"


def test_lab_finalize_mcp_routes_to_run_skill(monkeypatch):
    # The MCP handler must call run_skill("finalize_project", ...) and shape the
    # result for the host. Spy on run_skill (no provider/lab needed).
    from tools.mcp import bert_lab
    captured = {}

    def spy(skill_name, args, *, lab_path=None):
        captured["skill_name"] = skill_name
        captured["args"] = args
        return {"ok": True, "outputs": {"grade": "B", "signed_hash": "a" * 64,
                "artifact_path": "final.md", "gaps_path": "gaps.md", "ready": True},
                "errors": [], "steps_executed": []}

    monkeypatch.setattr(skill_runner, "run_skill", spy)
    srv = bert_lab.make_server()
    out = srv.tools["lab_finalize"].handler(
        {"objective": "X", "output_path": "final.md"})
    assert captured["skill_name"] == "finalize_project"
    assert captured["args"]["objective"] == "X"
    assert out["ok"] is True and out["grade"] == "B" and out["ready"] is True
    assert len(out["signed_hash"]) == 64


def test_lab_finalize_requires_objective_and_output():
    from tools.mcp import bert_lab
    srv = bert_lab.make_server()
    bad = srv.tools["lab_finalize"].handler({"objective": "X"})  # missing output_path
    assert bad["ok"] is False


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
    tests = [
        test_run_skill_executes_finalize_e2e,
        test_run_skill_unknown_skill,
        test_lab_finalize_mcp_tool_registered,
        test_lab_finalize_mcp_routes_to_run_skill,
        test_lab_finalize_requires_objective_and_output,
    ]
    for t in tests:
        mp = _MP()
        td = Path(tempfile.mkdtemp(prefix="bert_runner_"))
        try:
            params = inspect.signature(t).parameters
            kwargs = {}
            if "monkeypatch" in params:
                kwargs["monkeypatch"] = mp
            if "tmp" in params:
                kwargs["tmp"] = td
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
