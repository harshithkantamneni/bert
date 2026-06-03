"""Smoke: core/mcp_server.py — JSON-RPC MCP framework (was 64%).

Drives MCPServer.handle() in-process across every method: initialize,
notifications/initialized (no response), tools/list, tools/call (echo +
unknown-tool + nonce replay rejection), resources/list, prompts/list,
unknown method, and the notification path — plus register_tool, _err, and
make_echo_server. The stdio serve loop is integration-tier (skipped).
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import mcp_server as ms  # noqa: E402


def _rpc(method, params=None, req_id=1):
    msg = {"jsonrpc": "2.0", "method": method}
    if req_id is not None:
        msg["id"] = req_id
    if params is not None:
        msg["params"] = params
    return msg


def test_initialize_and_lists():
    srv = ms.make_echo_server()
    init = srv.handle(_rpc("initialize", {}))
    assert init["result"]["serverInfo"]["name"]
    tools = srv.handle(_rpc("tools/list"))
    assert any(t["name"] == "echo" for t in tools["result"]["tools"])
    assert srv.handle(_rpc("resources/list"))["result"]["resources"] == []
    assert srv.handle(_rpc("prompts/list"))["result"]["prompts"] == []


def test_tool_call_echo():
    srv = ms.make_echo_server()
    r = srv.handle(_rpc("tools/call", {"name": "echo", "arguments": {"text": "hi"}}))
    assert "result" in r


def test_tool_call_errors():
    srv = ms.make_echo_server()
    # unknown tool → an error comes back (the CODE is asserted precisely in
    # test_unknown_name_is_invalid_params; here we only assert it errors)
    unknown = srv.handle(_rpc("tools/call", {"name": "no_such_tool"}))
    assert "error" in unknown
    # unknown method → METHOD_NOT_FOUND
    bad = srv.handle(_rpc("frobnicate"))
    assert "error" in bad


def test_nonce_replay_rejected():
    import uuid
    srv = ms.make_echo_server()
    # unique nonce per run — the replay store PERSISTS seen nonces across
    # process restarts (anti-replay by design), so a hardcoded nonce would be
    # "already seen" on the 2nd-ever run and fail the first call.
    p = {"name": "echo", "arguments": {"text": "x"}, "_meta": {"nonce": uuid.uuid4().hex}}
    first = srv.handle(_rpc("tools/call", p, req_id=1))
    assert "result" in first
    # same nonce again → replay rejected (distinct error code)
    second = srv.handle(_rpc("tools/call", p, req_id=2))
    assert "error" in second


def test_notifications_return_none():
    srv = ms.make_echo_server()
    assert srv.handle(_rpc("notifications/initialized", req_id=None)) is None
    # an unknown-method notification (no id) → None, not an error
    assert srv.handle(_rpc("whatever", req_id=None)) is None


def test_register_tool_and_err():
    srv = ms.MCPServer(name="t", version="0.1")
    srv.register_tool("ping", description="ping", input_schema={"type": "object"},
                      handler=lambda a: {"pong": True})
    assert "ping" in srv.tools
    out = srv.handle(_rpc("tools/call", {"name": "ping", "arguments": {}}))
    assert "result" in out
    err = ms._err(5, ms.METHOD_NOT_FOUND, "nope")
    assert err["error"]["code"] == ms.METHOD_NOT_FOUND


def test_namespace_qualified_and_alias():
    # Sprint 4 A2 — a namespaced server emits qualified ids in list responses
    # (collision-safe across servers) but still resolves bare names (back-compat).
    srv = ms.MCPServer(name="bert-lab", version="0.1", namespace="bert.lab")
    srv.register_tool("lab_start", description="start a lab",
                      input_schema={"type": "object"}, handler=lambda a: {"ok": True})
    srv.register_resource(uri="bert://lab/x/seed", name="seed", description="d",
                          mime_type="text/markdown", reader=lambda: "body")
    srv.register_prompt("literature_survey", description="d",
                        arguments=[], builder=lambda a: [{"role": "user",
                                    "content": {"type": "text", "text": "go"}}])
    names = [t["name"] for t in srv.handle(_rpc("tools/list"))["result"]["tools"]]
    assert "bert.lab.lab_start" in names                      # qualified in list
    pnames = [p["name"] for p in srv.handle(_rpc("prompts/list"))["result"]["prompts"]]
    assert "bert.lab.literature_survey" in pnames
    # tools/call resolves BOTH the qualified id and the bare name
    assert "result" in srv.handle(_rpc("tools/call", {"name": "bert.lab.lab_start", "arguments": {}}))
    assert "result" in srv.handle(_rpc("tools/call", {"name": "lab_start", "arguments": {}}))
    # prompts/get resolves the qualified prompt id too
    assert "result" in srv.handle(_rpc("prompts/get", {"name": "bert.lab.literature_survey"}))


def test_unknown_name_is_invalid_params():
    # recheck 2026-05-28 — unknown tool/resource/prompt is a CLIENT error
    # (INVALID_PARAMS -32602), not INTERNAL_ERROR -32603 (which signals a
    # server bug). Asserts the CODE, not merely that an error came back.
    srv = ms.make_echo_server()
    srv.register_resource(uri="bert://x/y", name="y", description="d",
                          mime_type="text/plain", reader=lambda: "z")
    srv.register_prompt("p", description="d", arguments=[],
                        builder=lambda a: [{"role": "user",
                                            "content": {"type": "text", "text": "x"}}])
    for method, params in (("tools/call", {"name": "no_tool"}),
                           ("resources/read", {"uri": "bert://nope"}),
                           ("prompts/get", {"name": "no_prompt"})):
        out = srv.handle(_rpc(method, params))
        assert out["error"]["code"] == ms.INVALID_PARAMS, \
            f"{method} unknown-name should be INVALID_PARAMS, got {out['error']['code']}"


def main() -> int:
    tests = [
        test_initialize_and_lists,
        test_namespace_qualified_and_alias,
        test_unknown_name_is_invalid_params,
        test_tool_call_echo,
        test_tool_call_errors,
        test_nonce_replay_rejected,
        test_notifications_return_none,
        test_register_tool_and_err,
    ]
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
