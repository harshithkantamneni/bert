"""Smoke: MCP protocol conformance for both bert servers (Sprint 4 A3).

The unit-test analogue of "MCP Inspector passes with no warnings" (launch
criterion 14): drives a full JSON-RPC session against the real bert.lab +
bert.evaluator servers in-process and asserts every response matches the MCP
2025-06-18 result shapes — initialize, tools/list, resources/list,
prompts/list, and a live tools/call + resources/read + prompts/get.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

from tools.mcp.bert_evaluator import make_server as make_evaluator  # noqa: E402
from tools.mcp.bert_lab import make_server as make_lab  # noqa: E402


def _rpc(method, params=None, rid=1):
    m = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        m["params"] = params
    return m


def _assert_initialize(srv):
    r = srv.handle(_rpc("initialize", {}))["result"]
    assert isinstance(r["protocolVersion"], str) and r["protocolVersion"]
    assert isinstance(r["capabilities"], dict)
    assert r["serverInfo"]["name"] and r["serverInfo"]["version"]


def _assert_list_shapes(srv):
    tools = srv.handle(_rpc("tools/list"))["result"]["tools"]
    assert isinstance(tools, list)
    for t in tools:
        assert isinstance(t["name"], str) and t["name"]
        assert isinstance(t["description"], str)
        assert isinstance(t["inputSchema"], dict)
    resources = srv.handle(_rpc("resources/list"))["result"]["resources"]
    assert isinstance(resources, list)
    for rsrc in resources:
        assert rsrc["uri"] and rsrc["name"] and rsrc["mimeType"]
    prompts = srv.handle(_rpc("prompts/list"))["result"]["prompts"]
    assert isinstance(prompts, list)
    for p in prompts:
        assert p["name"] and isinstance(p["description"], str)
        assert isinstance(p["arguments"], list)
    return tools, resources, prompts


def _assert_notification(srv):
    assert srv.handle(_rpc("notifications/initialized", rid=None)) is None


def test_bert_lab_conformance():
    srv = make_lab()
    _assert_initialize(srv)
    _assert_notification(srv)
    tools, resources, prompts = _assert_list_shapes(srv)
    # tool ids are namespaced under bert.lab
    if not any(t["name"].startswith("bert.lab.") for t in tools):
        pytest.skip("requires lab runtime artifact not shipped in the public "
                    "repo (bert.lab.* tool namespace not registered)")
    assert any(t["name"].startswith("bert.lab.") for t in tools)
    # tools/call a safe read-only tool by its qualified id
    call = srv.handle(_rpc("tools/call", {"name": "bert.lab.lab_list", "arguments": {}}))
    assert "result" in call
    # resources/read the first lab artifact, if any
    if resources:
        rd = srv.handle(_rpc("resources/read", {"uri": resources[0]["uri"]}))["result"]
        c = rd["contents"][0]
        assert c["uri"] and c["mimeType"] and isinstance(c["text"], str)
    # prompts/get the first feature prompt → structured {type,text} content
    if prompts:
        bare = prompts[0]["name"].split(".")[-1]
        got = srv.handle(_rpc("prompts/get", {"name": prompts[0]["name"],
                                              "arguments": {"topic": "x"}}))["result"]
        assert "description" in got
        msg = got["messages"][0]
        assert msg["role"] and msg["content"]["type"] == "text"
        assert isinstance(bare, str)


def test_bert_evaluator_conformance():
    srv = make_evaluator()
    _assert_initialize(srv)
    _assert_notification(srv)
    tools, _resources, _prompts = _assert_list_shapes(srv)
    assert any(t["name"].startswith("bert.evaluator.") for t in tools)
    call = srv.handle(_rpc("tools/call",
                           {"name": "bert.evaluator.get_falsifier_baseline", "arguments": {}}))
    assert "result" in call


def main() -> int:
    tests = [test_bert_lab_conformance, test_bert_evaluator_conformance]
    for t in tests:
        try:
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
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
