"""Smoke: tools/mcp/bert_lab.py — the bert.lab MCP server (was 8%).

Builds the real MCPServer via make_server() and drives every tool handler
in-process: lab_list/status (+ missing lab), lab_start (temp lab,
heuristic classifier so no LLM), memory_search (use_vector=False → no
embedder), packet_export, and the error paths of lab_cycle / lab_resume /
lab_reshape (covering handler entry + arg resolution without a real LLM
dispatch). Side effects land in a temp ~/.bert/labs entry that is removed.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

bl = importlib.import_module("tools.mcp.bert_lab")

_TEMP_LAB = f"smoke_mcp_{abs(hash('bertlab')) % 99999}"


def test_make_server_registers_tools():
    srv = bl.make_server()
    assert srv is not None
    assert len(srv.tools) >= 6, f"expected ≥6 MCP tools, got {len(srv.tools)}"


def test_make_server_wires_resources_and_prompts():
    # Sprint 4 A1 — the bert.lab server exposes all 3 MCP primitives.
    srv = bl.make_server()
    # the 4 seed features are exposed as prompts
    assert {"literature_survey", "code_audit", "decision_memo", "refactor_plan"} <= set(srv.prompts)
    # prompts/get renders structured {type,text} content with the topic substituted
    got = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "prompts/get",
                      "params": {"name": "literature_survey", "arguments": {"topic": "vector DBs"}}})
    msg = got["result"]["messages"][0]
    assert msg["content"]["type"] == "text" and "vector DBs" in msg["content"]["text"]
    # resources/list is well-formed (lab artifacts when labs exist)
    res = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "resources/list"})["result"]["resources"]
    assert isinstance(res, list)


def test_resolve_lab_and_summary():
    # resolve a known lab + summarize it
    lp = bl._resolve_lab("test01")
    if lp is not None:
        summ = bl._lab_summary(lp)
        assert isinstance(summ, dict)


def test_lab_list_and_status():
    assert isinstance(bl._t_lab_list({}), dict)
    assert isinstance(bl._t_lab_list({"prefix": "test"}), dict)
    assert isinstance(bl._t_lab_status({"lab": "test01"}), dict)
    # unknown lab → handled, not a crash
    assert isinstance(bl._t_lab_status({"lab": "no_such_lab_xyz"}), dict)


def test_lab_start_temp_then_cleanup():
    res = bl._t_lab_start({
        "name": _TEMP_LAB,
        "mission": "Audit findings for stale claims; produce a ledger.",
        "use_llm_classifier": False,   # heuristic → no LLM call
    })
    assert isinstance(res, dict)
    # cleanup whatever it created
    created = Path.home() / ".bert" / "labs" / _TEMP_LAB
    if created.exists():
        shutil.rmtree(created, ignore_errors=True)


def test_memory_search_keyword_path():
    res = bl._t_memory_search({
        "lab": "test01", "query": "vector database", "k": 3, "use_vector": False,
    })
    assert isinstance(res, dict)


def test_packet_export():
    res = bl._t_packet_export({"lab": "test01"})
    assert isinstance(res, dict)


def test_error_paths_dont_crash():
    # lab_cycle on a nonexistent lab → entry + resolve + error (no dispatch)
    assert isinstance(bl._t_lab_cycle({"lab": "no_such_lab_xyz"}), dict)
    # resume with a bad token → error
    assert isinstance(bl._t_lab_resume({"token": "bad", "answer": "x"}), dict)
    # reshape with missing updates → error
    assert isinstance(bl._t_lab_reshape({"lab": "test01"}), dict)


def test_synthesize_tool_registered_and_proposes(monkeypatch=None):
    # Sprint 6 #30: lab_synthesize_tool runs synthesize -> sandbox -> propose,
    # returning a proposal id. The generated tool is NEVER registered here
    # (PI must /approve first). Provider + sandbox are stubbed (network/sandbox-free).
    srv = bl.make_server()
    assert "lab_synthesize_tool" in srv.tools

    import json as _json
    import tempfile as _tf

    from core import provider as prov
    from core import tool_registry
    from core import tool_synthesizer as ts
    from core.sandbox import SandboxResult, Tier
    from core.types import ProviderResponse

    safe_src = ("def my_synth_tool(**kwargs):\n"
                "    return {'ok': True}\n")
    saved_call, saved_validate = prov.call, ts.sandbox_validate
    saved_proposals, saved_pending = ts.PROPOSALS_PATH, ts.PENDING_DIR
    tmpd = _tf.mkdtemp()

    def fake_call(provider_name, messages, **kw):
        body = {"source": safe_src, "smoke_test": "assert my_synth_tool()['ok']\nprint('ok')\n"}
        return ProviderResponse(text=_json.dumps(body), tool_calls=[],
                                finish_reason="stop", usage_prompt_tokens=1,
                                usage_completion_tokens=1, usage_thinking_tokens=0,
                                usage_cached_tokens=0, model="stub",
                                provider=provider_name, elapsed_ms=1)

    def fake_validate(source, smoke, *, name, tier=None, work_dir=None, timeout_secs=30):
        return SandboxResult(stdout="ok\n", stderr="", exit_code=0,
                             elapsed_ms=1, tier_used=Tier.RESTRICTED)

    prov.call = fake_call
    ts.sandbox_validate = fake_validate
    ts.PROPOSALS_PATH = Path(tmpd) / "tools_pending_pi.md"
    ts.PENDING_DIR = Path(tmpd) / "pending"
    try:
        out = bl._t_lab_synthesize_tool({
            "name": "my_synth_tool",
            "description": "A tool that returns ok.",
        })
        assert out["ok"] is True
        assert out["proposal_id"].startswith("tool-")
        assert out["active"] is False  # not callable until PI approves
        assert tool_registry.get("my_synth_tool") is None
    finally:
        prov.call = saved_call
        ts.sandbox_validate = saved_validate
        ts.PROPOSALS_PATH = saved_proposals
        ts.PENDING_DIR = saved_pending
        shutil.rmtree(tmpd, ignore_errors=True)


def test_synthesize_tool_rejects_traversal_name():
    # name flows into a filename + function lookup — must reject non-identifiers.
    out = bl._t_lab_synthesize_tool({"name": "../../etc/passwd", "description": "x"})
    assert out["ok"] is False
    out2 = bl._t_lab_synthesize_tool({"name": "a b", "description": "x"})
    assert out2["ok"] is False


def test_approve_tool_registered_and_rejects_unknown():
    srv = bl.make_server()
    assert "lab_approve" in srv.tools
    out = bl._t_lab_approve({"proposal_id": "weird-xyz"})  # unknown prefix
    assert out["ok"] is False
    out2 = bl._t_lab_approve({})  # missing id
    assert out2["ok"] is False


def main() -> int:
    tests = [
        test_make_server_registers_tools,
        test_make_server_wires_resources_and_prompts,
        test_resolve_lab_and_summary,
        test_lab_list_and_status,
        test_lab_start_temp_then_cleanup,
        test_memory_search_keyword_path,
        test_packet_export,
        test_error_paths_dont_crash,
        test_synthesize_tool_registered_and_proposes,
        test_synthesize_tool_rejects_traversal_name,
        test_approve_tool_registered_and_rejects_unknown,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
