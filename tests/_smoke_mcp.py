"""Smoke test for the MCP layer (E.1) — client + installer + server.

Per FINAL_implementation_plan_amendment_2026-05-13.md §A1 E.1.

End-to-end: spawns bert's own echo MCP server, runs the initialize
handshake, lists tools, calls echo, closes cleanly. Plus unit tests
for the installer's JSON config schema.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import mcp_client, mcp_installer, mcp_server  # noqa: E402


def test_echo_server_handle_initialize_and_call_tool() -> None:
    """In-process MCPServer dispatch (no subprocess)."""
    srv = mcp_server.make_echo_server()
    # initialize handshake
    resp = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp["result"]["serverInfo"]["name"] == "bert-echo"
    assert "tools" in resp["result"]["capabilities"]
    # notifications/initialized → no response
    note_resp = srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert note_resp is None
    # tools/list
    resp = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = resp["result"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "echo"
    # tools/call
    resp = srv.handle({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "echo", "arguments": {"text": "hi"}},
    })
    content = resp["result"]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "hi" in content[0]["text"]


def test_unknown_method_returns_method_not_found() -> None:
    srv = mcp_server.make_echo_server()
    resp = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "nonsense", "params": {}})
    assert resp["error"]["code"] == mcp_server.METHOD_NOT_FOUND


def test_unknown_tool_returns_invalid_params() -> None:
    # recheck 2026-05-28 — an unknown tool name is a CLIENT error
    # (INVALID_PARAMS -32602), not INTERNAL_ERROR -32603 which signals a server
    # bug. This test previously asserted -32603 (it codified our own
    # non-conformant behavior); corrected alongside the mcp_server fix.
    srv = mcp_server.make_echo_server()
    resp = srv.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "doesnotexist", "arguments": {}},
    })
    assert resp["error"]["code"] == mcp_server.INVALID_PARAMS


def test_installer_load_save_roundtrip() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_mcp_install_")) / "registry.json"
    mcp_installer.install(
        "fetch", "uvx", ["mcp-server-fetch"],
        env={"FOO": "bar"}, description="HTTP fetch",
        path=tmp,
    )
    mcp_installer.install(
        "playwright", "npx", ["-y", "@playwright/mcp@latest"],
        path=tmp,
    )
    names = mcp_installer.list_configured(path=tmp)
    assert names == ["fetch", "playwright"]
    fetch = mcp_installer.load_spec("fetch", path=tmp)
    assert fetch is not None
    assert fetch.command == "uvx"
    assert fetch.args == ["mcp-server-fetch"]
    assert fetch.env == {"FOO": "bar"}
    assert fetch.description == "HTTP fetch"


def test_installer_uninstall() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_mcp_uninstall_")) / "registry.json"
    mcp_installer.install("fetch", "uvx", ["mcp-server-fetch"], path=tmp)
    assert mcp_installer.uninstall("fetch", path=tmp) is True
    assert mcp_installer.list_configured(path=tmp) == []
    # Idempotent: uninstalling a missing entry returns False, doesn't raise.
    assert mcp_installer.uninstall("fetch", path=tmp) is False


def test_installer_spawn_unknown_raises_filenotfound() -> None:
    tmp = Path(tempfile.mkdtemp()) / "registry.json"
    try:
        mcp_installer.spawn("does-not-exist", path=tmp)
    except FileNotFoundError as e:
        assert "does-not-exist" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_client_server_end_to_end_via_subprocess() -> None:
    """Spawn bert-echo as a real subprocess; do a full MCP handshake."""
    client = mcp_client.MCPClient.spawn(
        [sys.executable, str(LAB_ROOT / "lab.py"), "mcp", "bert-echo"],
        cwd=LAB_ROOT,
    )
    try:
        info = client.initialize()
        assert info["serverInfo"]["name"] == "bert-echo"
        tools = client.list_tools()
        assert any(t["name"] == "echo" for t in tools)
        result = client.call_tool("echo", {"text": "ping"})
        content = result.get("content", [])
        assert content and "ping" in content[0].get("text", "")
    finally:
        rc = client.close()
        assert rc == 0 or rc is None  # may return None if killed by close()


def main() -> int:
    tests = [
        test_echo_server_handle_initialize_and_call_tool,
        test_unknown_method_returns_method_not_found,
        test_unknown_tool_returns_invalid_params,
        test_installer_load_save_roundtrip,
        test_installer_uninstall,
        test_installer_spawn_unknown_raises_filenotfound,
        test_client_server_end_to_end_via_subprocess,
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
